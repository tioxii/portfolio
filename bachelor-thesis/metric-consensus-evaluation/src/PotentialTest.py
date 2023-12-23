from typing import Any
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import PotentialRework as pt

class ProbabilityOfConsensus():

    def __init__(self, path) -> None:
        self.path = path


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        start = time.time()
        
        print(self.path)
        result = pt.getPotential(self.path)
        print(result)

        end = time.time()
        print(end - start)

        return result

class MeanDistanceToMean():

    def __init__(self, path) -> None:
        self.path = path

    def __normalize(self, distance, max_distance):
        return distance / max_distance

    def __squared_distance(self, x, y, x_mean, y_mean):
        return (x - x_mean) ** 2 + (y - y_mean) ** 2
    
    def __mean_distance_to_mean(self):
        df1 = pd.read_csv(self.path)

        df1.columns = ['Round', 'x', 'y']
        df1 = df1.drop(df1[df1.Round > 0].index)
        _, x_mean, y_mean = df1.mean()

        df1['Squared Distance'] = df1.apply(lambda row: self.__squared_distance(row['x'], row['y'], x_mean, y_mean), axis=1)
        df1['Normalized Distance'] = df1.apply(lambda row: self.__normalize(row['Squared Distance'], df1['Squared Distance'].max()), axis=1)

        return df1['Normalized Distance'].mean()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.__mean_distance_to_mean()

class Potential:

    def __init__(self, cons_time_dir, pos_dir) -> None:
        self.cons_time_dir = cons_time_dir
        self.pos_dir = pos_dir

    def __consensus_time(self, path):
        df = pd.read_csv(path)
        df.columns = ['Round', 'Consensus Time']
        return df['Consensus Time'].mean()
    
    def __call__(self):
        df = pd.DataFrame()
        df['Consensus File'] = os.listdir(self.cons_time_dir)
        df['Positions File'] = os.listdir(self.pos_dir)

        df['Potential'] = df.apply(lambda row: ProbabilityOfConsensus(os.path.join(self.pos_dir, row['Positions File']))(), axis=1)
        df['Consensus Time'] = df.apply(lambda row: self.__consensus_time(os.path.join(self.cons_time_dir, row['Consensus File'])), axis=1)

        df.drop(columns=['Consensus File', 'Positions File'], inplace=True)

        return df

if __name__ == '__main__':
    cons_time_dir = 'results/potential/consensusTime'
    pos_dir = 'results/potential/positions'

    potential = Potential(cons_time_dir, pos_dir)
    df = potential()

    df['Index'] = df.index

    df['Consensus Time'] = df['Consensus Time'] / 100 


    fig, ax = plt.subplots()
    df.plot.scatter(x='Index', y='Potential', ax=ax)
    df.plot.scatter(x='Index', y='Consensus Time', ax=ax, color='orange')

    print(df.corr(method='pearson'))

    plt.show()

