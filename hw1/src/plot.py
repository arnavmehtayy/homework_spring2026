import csv
import matplotlib.pyplot as plt

def load():
    csv_path = "../exp/seed_42_20260205_184734/log.csv"
    
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'eval/mean_reward': float(row['eval/mean_reward']),
                'step': int(row['step'])
            })
    
    return data

def plot():
    data = load()
    plt.plot([d['step'] for d in data], [d['eval/mean_reward'] for d in data])
    plt.xlabel('Step')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward over Steps')
    plt.savefig('mean_reward.png')
    plt.close()

if __name__ == '__main__':
    plot()
    
