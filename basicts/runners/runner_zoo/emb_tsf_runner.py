from typing import Dict, Optional

import torch
import numpy as np
import os

from ..base_tsf_runner import BaseTimeSeriesForecastingRunner

import pdb

class EmbTimeSeriesForecastingRunner(BaseTimeSeriesForecastingRunner):
    """
    A Simple Runner for Time Series Forecasting with embedding and gate weight saving support.

    Args:
        cfg (Dict): Configuration dictionary.
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.forward_features = cfg['MODEL'].get('FORWARD_FEATURES', None)
        self.target_features = cfg['MODEL'].get('TARGET_FEATURES', None)
        self.target_time_series = cfg['MODEL'].get('TARGET_TIME_SERIES', None)
        
        # Flags for saving embeddings and gates
        self.save_embeddings = cfg.get('TEST', {}).get('SAVE_EMBEDDINGS', False)
        self.save_gate_weights = cfg.get('TEST', {}).get('SAVE_GATE_WEIGHTS', False)
        
        # Storage for embeddings and gate weights during testing
        self._embeddings_st = []
        self._embeddings_s = []
        self._embeddings_t = []
        self._gate_weights = []

    def preprocessing(self, input_data: Dict) -> Dict:
        """Preprocess data."""
        if self.scaler is not None:
            input_data['target'] = self.scaler.transform(input_data['target'])
            input_data['inputs'] = self.scaler.transform(input_data['inputs'])
        return input_data

    def postprocessing(self, input_data: Dict) -> Dict:
        """Postprocess data."""
        if self.scaler is not None and self.scaler.rescale:
            input_data['prediction'] = self.scaler.inverse_transform(input_data['prediction'])
            input_data['target'] = self.scaler.inverse_transform(input_data['target'])
            input_data['inputs'] = self.scaler.inverse_transform(input_data['inputs'])

        if self.target_time_series is not None:
            input_data['target'] = input_data['target'][:, :, self.target_time_series, :]
            input_data['prediction'] = input_data['prediction'][:, :, self.target_time_series, :]

        return input_data

    def forward(self, data: Dict, epoch: Optional[int] = None, iter_num: Optional[int] = None, 
                train: bool = True, return_embeddings: bool = False, **kwargs) -> Dict:
        """
        Performs the forward pass for training, validation, and testing. 

        Args:
            data (Dict): A dictionary containing 'target' and 'inputs'.
            epoch (int, optional): Current epoch number.
            iter_num (int, optional): Current iteration number.
            train (bool, optional): Indicates whether the forward pass is for training.
            return_embeddings (bool, optional): Whether to return embeddings and gate weights.

        Returns:
            Dict: A dictionary containing predictions, targets, and optionally embeddings.
        """
        data = self.preprocessing(data)

        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)
        future_data = self.to_running_device(future_data)
        batch_size, length, num_nodes, _ = future_data.shape

        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        if not train:
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # Forward pass with embedding extraction if needed
        model_return = self.model(
            history_data=history_data, 
            future_data=future_data_4_dec,
            batch_seen=iter_num, 
            epoch=epoch, 
            train=train,
            return_embeddings=return_embeddings
        )

        # Parse model return
        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}
        
        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)

        assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes], \
            "The shape of the output is incorrect. Ensure it matches [B, L, N, C]."

        model_return = self.postprocessing(model_return)

        return model_return

    def test(self, train_epoch: Optional[int] = None, save_metrics: bool = False, 
             save_results: bool = False) -> Dict:
        """
        Test process with embedding and gate weight saving.
        
        Args:
            train_epoch (Optional[int]): Current epoch if in training process.
            save_metrics (bool): Save the test metrics.
            save_results (bool): Save the test results.
        """
        # Clear storage
        self._embeddings_st = []
        self._embeddings_s = []
        self._embeddings_t = []
        self._gate_weights = []
        self._corr_features = []
        
        # Determine if we should extract embeddings
        extract_embeddings = self.save_embeddings or self.save_gate_weights
        
        # Call parent test method with modified forward
        from tqdm import tqdm
        import json
        
        for batch_idx, data in enumerate(tqdm(self.test_data_loader)):
            forward_return = self.forward(
                data, 
                epoch=None, 
                iter_num=None, 
                train=False,
                return_embeddings=extract_embeddings
            )

            loss = self.metric_forward(self.loss, forward_return)
            weight = self._get_metric_weight(forward_return['target'])
            self.update_epoch_meter('test/loss', loss.item(), weight)

            if not self.if_evaluate_on_gpu:
                pred = forward_return['prediction'].detach().cpu()
                target = forward_return['target'].detach().cpu()
            else:
                pred = forward_return['prediction']
                target = forward_return['target']
            
            # Store embeddings and gate weights if needed
            if extract_embeddings:
                if self.save_embeddings and 'embeddings' in forward_return:
                    self._embeddings_st.append(forward_return['embeddings']['st'].cpu().numpy())
                    self._embeddings_s.append(forward_return['embeddings']['s'].cpu().numpy())
                    self._embeddings_t.append(forward_return['embeddings']['t'].cpu().numpy())
                
                if self.save_gate_weights and 'gate_weights' in forward_return:
                    self._gate_weights.append(forward_return['gate_weights'].cpu().numpy())
                    self._corr_features.append(forward_return['corr_features'].cpu().numpy())
            if save_results:
                batch_data = {
                    'prediction': forward_return['prediction'].detach().cpu().numpy(),
                    'target': forward_return['target'].detach().cpu().numpy(),
                    'inputs': forward_return['inputs'].detach().cpu().numpy()
                }
                self._save_test_results(batch_idx, batch_data)

            # Evaluation on specific timesteps
            for i in self.evaluation_horizons:
                pred_h = pred[:, i, :, :]
                target_h = target[:, i, :, :]
                weight_h = self._get_metric_weight(target_h)

                for metric_name, metric_func in self.metrics.items():
                    if metric_name.lower() == 'mase':
                        continue
                    metric_val = self.metric_forward(metric_func, {'prediction': pred_h, 'target': target_h})
                    self.update_epoch_meter(f'test/{metric_name}@h{i+1}', metric_val.item(), weight_h)

            for metric_name, metric_func in self.metrics.items():
                metric_item = self.metric_forward(metric_func, {'prediction': pred, 'target': target})
                self.update_epoch_meter(f'test/{metric_name}', metric_item.item(), weight)

        # Save embeddings and gate weights after all batches
        if extract_embeddings:
            self._save_embeddings_and_gates()

        metrics_results = {}
        if save_metrics:
            metrics_results['overall'] = {k: self.meter_pool.get_value(f'test/{k}') for k in self.metrics.keys()}
            for i in self.evaluation_horizons:
                metrics_results[f'horizon_{i+1}'] = {k: self.meter_pool.get_value(f'test/{k}@h{i+1}') for k in self.metrics.keys()}

            with open(os.path.join(self.ckpt_save_dir, 'test_metrics.json'), 'w') as f:
                json.dump(metrics_results, f, indent=4)

        return metrics_results

    def _save_embeddings_and_gates(self):
        """Save accumulated embeddings and gate weights to files."""
        save_dir = os.path.join(self.ckpt_save_dir, 'embeddings')
        os.makedirs(save_dir, exist_ok=True)
        
        if self.save_embeddings:
            if self._embeddings_st:
                embeddings_st = np.concatenate(self._embeddings_st, axis=0)
                np.save(os.path.join(save_dir, 'embeddings_st.npy'), embeddings_st)
                self.logger.info(f'Saved ST embeddings with shape: {embeddings_st.shape}')
            
            if self._embeddings_s:
                embeddings_s = np.concatenate(self._embeddings_s, axis=0)
                np.save(os.path.join(save_dir, 'embeddings_s.npy'), embeddings_s)
                self.logger.info(f'Saved S embeddings with shape: {embeddings_s.shape}')
            
            if self._embeddings_t:
                embeddings_t = np.concatenate(self._embeddings_t, axis=0)
                np.save(os.path.join(save_dir, 'embeddings_t.npy'), embeddings_t)
                self.logger.info(f'Saved T embeddings with shape: {embeddings_t.shape}')
        
        if self.save_gate_weights and self._gate_weights:
            gate_weights = np.concatenate(self._gate_weights, axis=0)
            np.save(os.path.join(save_dir, 'gate_weights.npy'), gate_weights)
            self.logger.info(f'Saved gate weights with shape: {gate_weights.shape}')
            corr_features = np.concatenate(self._corr_features, axis=0)
            np.save(os.path.join(save_dir, 'corr_features.npy'), corr_features)
            self.logger.info(f'Saved correlated features with shape: {corr_features.shape}')

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select input features based on configuration."""
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select target features based on configuration."""
        data = data[:, :, :, self.target_features]
        return data

    def select_target_time_series(self, data: torch.Tensor) -> torch.Tensor:
        """Select target time series based on configuration."""
        data = data[:, :, self.target_time_series, :]
        return data