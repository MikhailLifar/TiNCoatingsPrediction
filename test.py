import numpy as np


def main():
    average_importance = np.zeros(5, dtype=[('feature_name', 'U20'), ('importance', float)])
    average_importance['importance'] = average_importance['importance'] + 1
    average_importance['feature_name'] = ['1', '01', '001', '001', '01']
    print(average_importance)


if __name__ == '__main__':
    main()
