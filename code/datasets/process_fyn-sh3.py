import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv("data/raw/fyn-sh3.csv", index_col=0)
    data = pd.DataFrame({'y': data['fitness_scaled'],
                         'y_var': data['sigma_scaled'] ** 2},
                        index=data.index.values)
    data.to_csv('data/processed/fyn-sh3.csv')
    