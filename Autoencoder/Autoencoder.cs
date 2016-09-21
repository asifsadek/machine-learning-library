using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace Machine_Learning.Autoencoder {
    public class Autoencoder {
        public static Random rnd = new Random();
        public const double SPARSITY_COST = 0.001;
        public const double SPARSITY_TARGET = 0.15;
        public const double SPARSITY_ESTIMATION = 0.999;

        static Random rand = new Random();

        public double corruptionLevel;
        public List<Layer> layers;
        public double totalCost, iterations;
        public Queue<double> costList;

        public Layer decoder;

        public Autoencoder (double corruptionLevel) {
            this.layers = new List<Layer>();
            this.costList = new Queue<double>();
            this.corruptionLevel = corruptionLevel;
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

        public void stack (Layer encoder, Layer decoder) {
            layers.Insert((layers.Count + 1) / 2, encoder);
            if (this.decoder != null)
                layers.Insert((layers.Count + 1) / 2 + 1, this.decoder);
            this.decoder = decoder;

            for (int i = 1; i < layers.Count; i++) {
                Layer prevLayer = layers[i - 1];
                if ((layers.Count) / 2 <= i && i <= (layers.Count + 1) / 2)
                    layers[i].BindTo(ref prevLayer);
                else
                    layers[i].RebindTo(ref prevLayer);
            }
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
            double[] encoderInput = null;
            iterations++;

            ((InputLayer)layers[0]).forwardPropagate(input);
            
            for (int i = 0; i <= (layers.Count + 1) / 2; i++) {
                layers[i].forwardPropagate();
                if (i == layers.Count / 2 - 1) {
                    encoderInput = new double[layers[i].size];
                    for (int j = 0; j < layers[i].size; j++) {
                        encoderInput[j] = layers[i].neurons[j].activated;
                        if (rnd.NextDouble() < corruptionLevel) {
                            layers[i].neurons[j].activated = 0;
                        }
                    }
                }
            }

            double currCost = ((FullyConnectedLayer)layers[(layers.Count + 1) / 2]).backPropagate(encoderInput);

            totalCost += currCost;
            costList.Enqueue(currCost);

            for (int i = (layers.Count + 1) / 2; i >= (layers.Count + 1) / 2 - 1; i--)
                layers[i].backPropagate(learningRate, i == (layers.Count + 1) / 2 - 1);
        }

        public double[] getEncodedValues (double[] input) {
            ((InputLayer)layers[0]).forwardPropagate(input);
            for (int i = 0; i <= layers.Count / 2; i++)
                layers[i].forwardPropagate();
            int encodedSize = layers[layers.Count / 2].size;
            double[] ret = new double[encodedSize];

            for (int i = 0; i < encodedSize; i++)
                ret[i] = layers[layers.Count / 2].neurons[i].activated;

            return ret;
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
