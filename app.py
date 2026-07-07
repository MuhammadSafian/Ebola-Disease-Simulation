
from flask import Flask, render_template, jsonify, request
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import EbolaDataLoader
from slp import SingleLayerPerceptron
from abm import EbolaDiseaseModel

app = Flask(__name__)
data_loader = None
slp = None
model = None

def init_globals():
    global data_loader, slp
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ebola_2014_2016_clean.csv')
    data_loader = EbolaDataLoader(csv_path)
    slp = SingleLayerPerceptron(n_inputs=4, learning_rate=0.1)
    X, y = data_loader.get_slp_training_data()
    if len(X) > 0:
        slp.train(X, y, epochs=100)
        print(f"[SLP] Trained: {len(X)} samples, acc={slp.get_info()['accuracy']:.4f}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/init', methods=['POST'])
def api_init():
    global model
    data = request.get_json() or {}
    n_agents = max(50, min(10000, int(data.get('population', 5000))))
    country = data.get('country', 'Guinea')
    infection_rate = max(0.01, min(1.0, float(data.get('infection_rate', 0.15))))
    initial_infected = max(1, min(n_agents // 10, int(data.get('initial_infected', 3))))
    model = EbolaDiseaseModel(n_agents=n_agents, infection_rate=infection_rate,
                              country=country, initial_infected=initial_infected,
                              slp=slp, data_loader=data_loader)
    return jsonify({'status': 'ok', 'state': model.get_full_state(),
                    'slp_info': slp.get_info() if slp else None,
                    'country_params': model.country_params})

@app.route('/api/step', methods=['POST'])
def api_step():
    global model
    if model is None:
        return jsonify({'error': 'Model not initialized'}), 400
    data = request.get_json() or {}
    for _ in range(max(1, min(10, int(data.get('steps', 1))))):
        model.step()
    return jsonify({'status': 'ok', 'state': model.get_full_state()})

@app.route('/api/reset', methods=['POST'])
def api_reset():
    global model
    model = None
    return jsonify({'status': 'ok'})

@app.route('/api/ebola-data')
def api_ebola_data():
    country = request.args.get('country', 'Guinea')
    return jsonify({'country': country,
                    'timeseries': data_loader.get_country_timeseries(country),
                    'daily_new_cases': data_loader.get_daily_new_cases(country)})

@app.route('/api/slp-info')
def api_slp_info():
    if slp is None:
        return jsonify({'error': 'SLP not trained'}), 500
    return jsonify(slp.get_info())

@app.route('/api/countries')
def api_countries():
    return jsonify({'countries': data_loader.get_countries(),
                    'stats': data_loader.get_summary_stats()})

@app.route('/api/agents')
def api_agents():
    if model is None:
        return jsonify({'error': 'Model not initialized'}), 400
    return jsonify({'top_agents': model.get_top_agents(10)})

if __name__ == '__main__':
    print("=" * 50)
    print("  Ebola ABM + SLP Disease Spread Simulation")
    print("=" * 50)
    init_globals()
    print(f"\n[SERVER] http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
