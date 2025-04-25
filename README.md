# GNN-Based CFD Surrogate with LBM Simulator

This project demonstrates a lightweight, self-contained system for **learning and optimizing CFD flow metrics** using a **Graph Neural Network (GNN)** surrogate model trained on **Lattice Boltzmann Method (LBM)** simulations.

It was developed as part of a personal upskilling effort to transition into applied machine learning for engineering and physical systems. The system supports both **live simulation** using the LBM and **real-time prediction** via a trained GNN, with an integrated optimization loop to explore geometry-performance trade-offs.

## Key Features

- **GNN surrogate model** trained to predict drag, lift, and mean wake vorticity for 2D shapes
- **LBM simulation engine** for generating ground truth velocity fields
- **Integrated UI** for drawing shapes, visualizing flow fields, and running optimization
- **Model vs simulation comparison** to assess surrogate accuracy
- **Optimization loop** that adjusts shape to target performance metrics
- **FastAPI + Streamlit-style interface** for real-time interactivity
- **Containerized setup** for deployment and reproducibility

## Demo Screenshot

![Surrogate vs LBM Screenshot](doc/screenshot.png)
![Training performance](doc/r2_wings.png)
![LBM animation](doc/velocity.gif)

## Technologies Used

- PyTorch Geometric (GNN training)
- NumPy / Matplotlib (geometry and plotting)
- Custom LBM solver (2D CFD on a regular grid)
- Streamlit / FastAPI (interactive UI)
- MLFlow (experiment tracking)
- Docker (optional containerization)

## How It Works

1. User draws or loads a 2D shape.
2. The surrogate model predicts drag, lift, and wake metrics instantly.
3. The shape is passed to the LBM solver for simulation and comparison.
4. An optimization loop adjusts geometry to minimize drag or wake vorticity.
5. All steps are visualized with velocity fields and numerical outputs.

## Motivation

This project reflects my interest in **bringing machine learning to bear on physical systems**—especially in areas like simulation acceleration, surrogate modeling, and closed-loop optimization. It draws inspiration from work at leading scientific ML companies and showcases what can be done at small scale with thoughtful integration of classical and modern techniques.

## Future Directions

- Active learning loop to retrain the surrogate on uncertain samples
- Reinforcement learning agent to explore geometric design space
- Extension to 3D shapes and more complex fluid domains
- Integration with physics-informed loss functions or PDE solvers

## License

MIT License
