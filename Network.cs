using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace Machine_Learning {

    public class Network {
        public const int RELU = 0;
        public const int TANH = 1;
        public const int SIGMOID = 2;
        public const int LINEAR = 3;

        public List<Layer> layers;
        public double correct, total, totalCost, iterations;
        public Queue<bool> correctList;
        public Queue<double> costList;

        public Network () {
            layers = new List<Layer>();
            correctList = new Queue<bool>();
            costList = new Queue<double>();
        }

        public Network (string path) {
            layers = new List<Layer>();
            correctList = new Queue<bool>();
            costList = new Queue<double>();
            load(path);
        }

        public void addLayer (Layer layer) {
            if (layers.Count > 0) {
                Layer prevLayer = layers.Last();
                layer.BindTo(ref prevLayer);
            }
            layers.Add(layer);
        }

        public int predict (double[] input) {
            for (int i = 0; i < layers.Count; i++) {
                if (i == 0)
                    ((InputLayer1D)layers[i]).forwardPropagate(input);
                layers[i].forwardPropagate();
            }
            return maxIndex(layers.Last().neurons);
        }

        public int predict (double[,,] input) {
            for (int i = 0; i < layers.Count; i++) {
                if (i == 0)
                    ((InputLayer2D)layers[i]).forwardPropagate(input);
                layers[i].forwardPropagate();
            }
            return maxIndex(layers.Last().neurons);
        }

        public void train (double[] input, int answer, double learningRate) {
            total++;
            iterations++;

            int outputSize = ((Layer1D)layers.Last()).size;
            double[] error = new double[outputSize];

            for (int i = 0; i < outputSize; i++)
                error[i] = (answer == i ? 1 : -1);

            for (int i = 0; i < layers.Count; i++) {
                if (i == 0)
                    ((InputLayer1D)layers[i]).forwardPropagate(input);
                layers[i].forwardPropagate();
            }

            for (int i = layers.Count - 1; i >= 0; i--) {
                if (i == layers.Count - 1)
                    ((OutputLayer)layers[i]).backPropagate(error);

                layers[i].backPropagate(learningRate);
                if (i == layers.Count - 1) {
                    if (maxIndex(layers[i].neurons) == answer) {
                        correct++;
                        correctList.Enqueue(true);
                    } else {
                        correctList.Enqueue(false);
                    }

                    double cost = 0;
                    for (int j = 0; j < outputSize; j++) {
                        double currCost = layers[i].neurons[j].error * layers[i].neurons[j].error / (layers[i].neurons[j].getDerivative() + 1e-8F);
                        cost += currCost * currCost;
                    }
                    totalCost += cost;
                    costList.Enqueue(cost);
                }
            }
        }

        public void train (double[,,] input, int answer, double learningRate) {
            total++;
            iterations++;

            int outputSize = ((Layer1D)layers.Last()).size;
            double[] error = new double[outputSize];

            for (int i = 0; i < outputSize; i++)
                error[i] = (answer == i ? 1 : -1);

            ((InputLayer2D)layers[0]).forwardPropagate(input);

            for (int i = 0; i < layers.Count; i++)
                layers[i].forwardPropagate();


            double currCost = ((OutputLayer)layers[layers.Count - 1]).backPropagate(error);
            totalCost += currCost;
            costList.Enqueue(currCost);
            if (maxIndex(layers[layers.Count - 1].neurons) == answer) {
                correct++;
                correctList.Enqueue(true);
            } else {
                correctList.Enqueue(false);
            }

            for (int i = layers.Count - 1; i >= 0; i--)
                layers[i].backPropagate(learningRate);
        }

        private int maxIndex (Neuron[] output) {
            int max = 0;
            for (int i = 1; i < output.GetLength(0); i++)
                if (output[i].activated > output[max].activated)
                    max = i;
            return max;
        }

        public void save (string filepath) {
            using (StreamWriter writer = new StreamWriter(filepath)) {
                writer.WriteLine(ToString());
            }
        }

        private void load (string filepath) {
            using (StreamReader reader = new StreamReader(filepath)) {
                iterations = int.Parse(reader.ReadLine());
                int numberOfLayers = int.Parse(reader.ReadLine());

                for (int i = 0; i < numberOfLayers; i++) {
                    Type layerType = Type.GetType(reader.ReadLine());
                    ConstructorInfo constructor = layerType.GetConstructor(new Type[2] { typeof(StreamReader), typeof(Layer) });
                    layers.Add((Layer)constructor.Invoke(new object[] { reader, i == 0 ? null : layers.Last() }));
                }
            }
        }

        public override String ToString () {
            StringBuilder sb = new StringBuilder();

            sb.Append(iterations + "\n");
            sb.Append(layers.Count + "\n");

            for (int i = 0; i < layers.Count; i++) {
                sb.Append(layers[i].ToString() + "\n");
            }

            return sb.ToString();
        }
    }
}
