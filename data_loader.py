import os
import numpy as np
import pandas as pd

class EbolaDataLoader:
    """Load and process the clean Ebola dataset using pandas."""

    TARGET_COUNTRIES = ['Guinea', 'Liberia', 'Sierra Leone']

    # Country-specific epidemiological parameters for calibration
    COUNTRY_PARAMS = {
        'Guinea': {
            'infection_rate': 0.15,
            'cfr': 0.66,
            'healthcare_access': 0.30,
            'population_density': 0.45,
            'incubation_mean': 9,
        },
        'Liberia': {
            'infection_rate': 0.20,
            'cfr': 0.74,
            'healthcare_access': 0.25,
            'population_density': 0.50,
            'incubation_mean': 8,
        },
        'Sierra Leone': {
            'infection_rate': 0.18,
            'cfr': 0.39,
            'healthcare_access': 0.28,
            'population_density': 0.55,
            'incubation_mean': 10,
        }
    }

    def __init__(self, csv_path=None):
        if csv_path is None or not os.path.exists(csv_path):
            base = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(base, 'ebola_2014_2016_clean.csv')

        self.csv_path = csv_path
        self.df = None
        self._load_and_clean_data()

    def _load_and_clean_data(self):
        """Read and clean the new ebola_2014_2016_clean.csv dataset."""
        df = pd.read_csv(self.csv_path)
        
        # Filter for the target countries as requested by the user
        df = df[df["Country"].isin(self.TARGET_COUNTRIES)].copy()
        
        # Standardize date and convert cases/deaths to numeric
        df['Date'] = pd.to_datetime(df['Date'])
        
        cases_col = "Cumulative no. of confirmed, probable and suspected cases"
        deaths_col = "Cumulative no. of confirmed, probable and suspected deaths"
        
        df[cases_col] = pd.to_numeric(df[cases_col], errors='coerce').fillna(0.0)
        df[deaths_col] = pd.to_numeric(df[deaths_col], errors='coerce').fillna(0.0)
        
        # Sort values chronologically
        df = df.sort_values(by='Date').reset_index(drop=True)
        self.df = df

    def get_countries(self):
        """Return the target countries."""
        return self.TARGET_COUNTRIES[:]

    def get_country_timeseries(self, country):
        """Return cumulative cases and deaths for a country."""
        country_df = self.df[self.df['Country'] == country]
        if country_df.empty:
            return {'dates': [], 'cases': [], 'deaths': []}

        dates = country_df['Date'].dt.strftime('%Y-%m-%d').tolist()
        
        cases_col = "Cumulative no. of confirmed, probable and suspected cases"
        deaths_col = "Cumulative no. of confirmed, probable and suspected deaths"
        
        cases = country_df[cases_col].tolist()
        deaths = country_df[deaths_col].tolist()

        return {
            'dates': dates,
            'cases': cases,
            'deaths': deaths
        }

    def get_daily_new_cases(self, country):
        """Calculate daily new cases from cumulative timeseries."""
        ts = self.get_country_timeseries(country)
        dates = ts['dates']
        cases = ts['cases']

        new_cases = [0]
        for i in range(1, len(cases)):
            diff = cases[i] - cases[i - 1]
            new_cases.append(max(0, diff))

        return {
            'dates': dates,
            'new_cases': new_cases
        }

    def get_slp_training_data(self):
        """Generate synthetic agent-level features for Perceptron training."""
        X_list = []
        y_list = []

        np.random.seed(42)
        for _ in range(1500):
            country = np.random.choice(self.TARGET_COUNTRIES)
            params = self.COUNTRY_PARAMS[country]

            age_norm = np.random.uniform(0, 1)
            contacts_norm = np.random.beta(1, 5)
            healthcare_access = np.clip(np.random.normal(params['healthcare_access'], 0.1), 0, 1)
            crowding = np.random.uniform(0, 1)

            if contacts_norm < 0.05:
                risk_score = 0.0
            else:
                base_risk = params['infection_rate'] * 5.0
                risk_score = base_risk * (0.5 * contacts_norm + 0.3 * crowding + 0.2 * age_norm)
                risk_score *= (1.0 - healthcare_access * 0.5)

            risk_score += np.random.normal(0, 0.05)

            y_list.append(1 if risk_score > 0.4 else 0)
            X_list.append([age_norm, contacts_norm, healthcare_access, crowding])

        return np.array(X_list, dtype=np.float64), np.array(y_list, dtype=np.float64)

    def get_abm_params(self, country='Guinea'):
        """Return country parameters for agent modeling."""
        if country not in self.COUNTRY_PARAMS:
            country = 'Guinea'

        params = self.COUNTRY_PARAMS[country].copy()
        ts = self.get_country_timeseries(country)
        
        if ts['cases']:
            params['total_cases'] = int(max(ts['cases']))
            params['total_deaths'] = int(max(ts['deaths'])) if ts['deaths'] else 0
            params['actual_cfr'] = params['total_deaths'] / max(params['total_cases'], 1)
        else:
            params['total_cases'] = 0
            params['total_deaths'] = 0
            params['actual_cfr'] = params['cfr']

        return params

    def get_summary_stats(self):
        """Return brief overview stats of each country for the dashboard."""
        stats = {}
        for country in self.TARGET_COUNTRIES:
            ts = self.get_country_timeseries(country)
            total_cases = int(max(ts['cases'])) if ts['cases'] else 0
            total_deaths = int(max(ts['deaths'])) if ts['deaths'] else 0
            stats[country] = {
                'total_cases': total_cases,
                'total_deaths': total_deaths,
                'cfr': round(total_deaths / max(total_cases, 1) * 100, 1) if total_cases > 0 else 0.0,
                'data_points': len(ts['dates']),
                'date_range': f"{ts['dates'][0]} to {ts['dates'][-1]}" if ts['dates'] else 'N/A'
            }
        return stats

if __name__ == '__main__':
    # Test loader directly
    loader = EbolaDataLoader()
    print("Clean loader loaded successfully!")
    print("Target countries in dataset:", loader.get_countries())
    print("\nSummary statistics from ebola_2014_2016_clean.csv:")
    for country, s in loader.get_summary_stats().items():
        print(f"  {country}: {s['total_cases']} cases, {s['total_deaths']} deaths, CFR={s['cfr']}% (from {s['data_points']} dates)")
