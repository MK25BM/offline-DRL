# offline-DRL

A comprehensive offline reinforcement learning (RL) pipeline for Type 1 Diabetes (T1D) glucose control simulation, implemented with **custom PyTorch-based algorithms** to avoid dependency conflicts.

## Features

- **No d3rlpy dependency**: Custom implementation of offline RL algorithms
- **Python 3.12 compatible**: No scipy/scikit-learn conflicts
- **Simplified dependencies**: Only numpy, gymnasium, torch, and minari
- **Custom algorithms**: DQN and Behavioral Cloning implemented from scratch
- **Minari integration**: Dataset management without scikit-learn

## Dependencies

```bash
pip install numpy gymnasium torch minari
```

## Algorithms Implemented

1. **Offline DQN**: Deep Q-Network trained on fixed offline dataset
2. **Behavioral Cloning**: Supervised learning to imitate behavior policy

## Usage

Open the Jupyter notebook `Github_Offline_DRL.ipynb` in Google Colab or locally:

1. **Environment Setup**: Mock T1D environment compatible with Gymnasium
2. **Data Collection**: Collect episodes using behavior policy
3. **Algorithm Training**: Train offline RL algorithms on collected data
4. **Evaluation**: Test trained policies and compare performance
5. **Model Persistence**: Save and load trained models

## Key Improvements

- Removed all d3rlpy, scipy, and scikit-learn dependencies
- Implemented custom replay buffer for offline training
- PyTorch-based neural networks for Q-learning and policy learning
- Compatible with Python 3.12+ without numpy.char issues
- All original functionality maintained with cleaner implementation
