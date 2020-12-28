from main import MNIST3dClassifier
import yaml

def main():
    config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
    print(config)
    
    clf = MNIST3dClassifier(config)
    clf.train()
    clf.test()


if __name__ == "__main__":
    main()
