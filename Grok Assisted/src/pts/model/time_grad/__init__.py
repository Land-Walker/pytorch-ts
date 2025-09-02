"""__init__.py

Package initialization for TimeGrad module.
Imports advanced TimeGrad classes for easy access in your project.
"""
from .diffusion import Diffusion, DiffusionConfig, GaussianDiffusion
from .epsilon_theta import EpsilonTheta, EpsilonThetaConfig, create_timegrad_epsilon_theta
from .time_grad_network import (
    TimeGradTrainingNetwork, 
    TimeGradPredictionNetwork, 
    TimeGradConfig
)
from .time_grad_estimator import TimeGradEstimator, TimeGradEstimatorConfig
from .time_grad_predictor import TimeGradPredictor, TimeGradPredictorFactory
from src.pts.trainer import Trainer


__all__ = [
    # Diffusion components
    'Diffusion',
    'DiffusionConfig', 
    'GaussianDiffusion',
    
    # Advanced denoising network
    'EpsilonTheta',
    'EpsilonThetaConfig',
    'create_timegrad_epsilon_theta',
    
    # TimeGrad networks
    'TimeGradTrainingNetwork',
    'TimeGradPredictionNetwork',
    'TimeGradConfig',
    
    # Estimator
    'TimeGradEstimator',
    'TimeGradEstimatorConfig',
    
    # Predictor
    'TimeGradPredictor',
    'TimeGradPredictorFactory',
]

if __name__ == "__main__":
    print("Advanced TimeGrad module initialized successfully!")
    print("Available components:")
    for component in __all__:
        print(f"  - {component}")