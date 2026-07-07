# Ebola Disease Spread Simulation (ABM + ANN)

This project is a comprehensive web-based simulation platform designed to model the spread of the Ebola virus. It utilizes historical epidemiological data from the 2014-2016 Ebola outbreak and combines **Agent-Based Modeling (ABM)** with a **Single Layer Perceptron (SLP) Artificial Neural Network** to simulate and analyze the disease's progression dynamics across different populations.

## Key Features

- **Agent-Based Modeling (ABM):** Simulates individual agents in a population, tracking their states (Susceptible, Infected, Recovered, Dead) based on configurable infection rates and interactions.
- **Artificial Neural Network (ANN):** Integrates a Single Layer Perceptron (SLP) trained on real-world data to classify and predict outbreak severity based on country demographics and initial conditions.
- **Real-World Data Integration:** Processes and utilizes the clean `ebola_2014_2016` dataset to inform simulation parameters.
- **Interactive Web Dashboard:** Built with Flask (Backend) and HTML/JS (Frontend) to allow users to dynamically adjust population size, initial infections, infection rates, and the target country.
- **Real-Time Visualization:** Dynamically renders the simulation states and epidemiological curves over time.

## Tech Stack

- **Backend:** Python 3, Flask
- **Machine Learning & Modeling:** Python (Custom implementations for ABM and SLP)
- **Frontend:** HTML5, CSS3, Vanilla JavaScript
- **Data:** CSV (Ebola 2014-2016 Outbreak Dataset)

## Project Structure

- `app.py`: The main Flask application server and API endpoints.
- `abm.py`: Contains the logic for the Agent-Based Model, managing agent states and interactions.
- `slp.py`: The Single Layer Perceptron implementation for classification.
- `data_loader.py`: Handles loading, parsing, and cleaning of the Ebola dataset.
- `static/`: Contains static assets like CSS stylesheets and JavaScript files for the UI.
- `templates/`: Contains HTML templates for the Flask application.

## How to Run

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd ABM--ANN-PROJECT
   ```

2. **Install Dependencies:**
   Ensure you have Python installed. You can install the required packages (like Flask) using pip:
   ```bash
   pip install flask
   # Add any other dependencies like pandas/numpy if they are used in data_loader.py
   ```

3. **Run the Server:**
   ```bash
   python app.py
   ```

4. **Access the Application:**
   Open your web browser and navigate to `http://localhost:5000`

## Author
[Your Name/Username]
