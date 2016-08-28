using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Machine_Learning.Neural_Network {
    public class OutputLayer : Layer1D {

        public OutputLayer (int size, int type) {
            if (type != Network.SIGMOID && type != Network.SOFTMAX && type != Network.LINEAR)
                throw new System.ArgumentException();
            this.size = size;
            this.type = type;
            this.neurons = new Neuron[size];
        }

        public OutputLayer (StreamReader reader, Layer prev) {
            string[] data = reader.ReadLine().Split();
            this.size = int.Parse(data[0]);
            this.type = int.Parse(data[1]);

            this.neurons = new Neuron[size];

            BindTo(ref prev);

            for (int i = 0; i < size; i++) {
                string[] input = reader.ReadLine().Split();
                neurons[i].weights.bias = double.Parse(input[0]);
                for (int j = 0; j < neurons[i].weights.val.GetLength(0); j++)
                    neurons[i].weights.val[j] = double.Parse(input[j + 1]);
            }
        }

        public override void BindTo (ref Layer layer) {
            prevLayer = layer;

            for (int i = 0; i < size; i++) {
                neurons[i] = new Neuron(prevLayer.size, this.type);
                neurons[i].link(prevLayer.neurons);
            }
        }

        public override void forwardPropagate () {
            if (type == Network.SIGMOID) {
                for (int i = 0; i < size; i++)
                    neurons[i].forwardPropagate();

            } else if (type == Network.SOFTMAX) {
                for (int i = 0; i < size; i++)
                    neurons[i].forwardPropagate();

                double sum = 0;
                for (int i = 0; i < size; i++)
                    sum += neurons[i].activated;

                for (int i = 0; i < size; i++)
                    neurons[i].activated /= sum;
            } else if (type == Network.LINEAR) {
                for (int i = 0; i < size; i++)
                    neurons[i].forwardPropagate();
            }
        }

        public override void backPropagate (double learningRate) {
            for (int i = 0; i < size; i++)
                neurons[i].backPropagateError(true);


            for (int i = 0; i < size; i++) {
                neurons[i].backPropagateRegularize(learningRate);
                neurons[i].backPropagateWeights(learningRate);
            }
        }

        public double backPropagate (double[] error) {
            double ret = 0;

            if (type == Network.SIGMOID) {
                for (int i = 0; i < size; i++) {
                    neurons[i].error = (neurons[i].activated - error[i]);
                    ret -= error[i] * Math.Log(neurons[i].activated) + (1 - error[i]) * Math.Log(1 - neurons[i].activated);
                }
            } else if (type == Network.SOFTMAX) {
                for (int i = 0; i < size; i++) {
                    neurons[i].error = (neurons[i].activated - error[i]);
                    ret -= error[i] * Math.Log(neurons[i].activated);
                }
            } else if (type == Network.LINEAR) {
                for (int i = 0; i < size; i++) {
                    neurons[i].error = (neurons[i].activated - error[i]) * neurons[i].getDerivative();
                    ret += (neurons[i].activated - error[i]) * (neurons[i].activated - error[i]);
                }
            }

            return ret;
        }

        public override String ToString () {
            StringBuilder sb = new StringBuilder();
            sb.Append(String.Format("{0}\n{1} {2}", this.GetType().FullName, size, type));
            for (int i = 0; i < size; i++) {
                sb.Append("\n" + neurons[i].weights.bias);
                for (int j = 0; j < neurons[i].weights.val.GetLength(0); j++)
                    sb.Append(" " + neurons[i].weights.val[j]);
            }
            return sb.ToString();
        }
    }
}
