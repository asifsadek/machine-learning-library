using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace Machine_Learning.Autoencoder {
    public class Autoencoder {
        public const double SPARSITY_COST = 0.01;
        public const double SPARSITY_TARGET = 0.05;

        static Random rand = new Random();

        public List<Layer> layers;
        public double totalCost, iterations;
        public Queue<double> costList;

        public Autoencoder () {
            layers = new List<Layer>();
            costList = new Queue<double>();
        }

        public Autoencoder (string path) {
            layers = new List<Layer>();
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

        public double[] predict (double[] input) {
            double[] ret = new double[input.GetLength(0)];

            ((InputLayer)layers[0]).forwardPropagate(input);

            for (int i = 0; i < layers.Count; i++)
                layers[i].forwardPropagate();

            for (int i = 0; i < layers.Last().size; i++)
                ret[i] = layers.Last().neurons[i].activated;

            return ret;
        }

        public void train (double[] input, int answer, double learningRate) {
            iterations++;

            int outputSize = ((Layer)layers.Last()).size;

            ((InputLayer)layers[0]).forwardPropagate(input);

            for (int i = 0; i < layers.Count; i++)
                layers[i].forwardPropagate();
            
            double currCost = ((OutputLayer)layers[layers.Count - 1]).backPropagate(input);

            totalCost += currCost;
            costList.Enqueue(currCost);

            for (int i = layers.Count - 1; i >= 0; i--)
                layers[i].backPropagate(learningRate);
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
