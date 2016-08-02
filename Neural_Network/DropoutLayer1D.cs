using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Machine_Learning.Neural_Network {

    public class DropoutLayer1D : Layer1D {

        static Random rand = new Random();

        // probability to keep neuron
        public double prob;

        public DropoutLayer1D (double prob) {
            this.prob = prob;
        }

        public DropoutLayer1D (StreamReader reader, Layer prev) {
            string[] data = reader.ReadLine().Split();
            this.prob = double.Parse(data[0]);

            BindTo(ref prev);
        }

        public override void BindTo (ref Layer layer) {
            prevLayer = layer;

            this.neurons = new Neuron[prevLayer.size];
            this.size = prevLayer.size;

            for (int i = 0; i < size; i++) {
                neurons[i] = new Neuron(1, this.type);
                neurons[i].link(new Neuron[]{prevLayer.neurons[i]});
            }
        }

        public override void forwardPropagate () {
            for (int i = 0; i < size; i++) {
                if (rand.NextDouble() < prob)
                    neurons[i].val = neurons[i].activated = neurons[i].prev[0].activated;
                else
                    neurons[i].val = neurons[i].activated = 0;
            }
        }

        public void forwardPropagateTest () {
            for (int i = 0; i < size; i++)
                neurons[i].val = neurons[i].activated = neurons[i].prev[0].activated * prob;
        }

        public override void backPropagate (double learningRate) {
            for (int i = 0; i < size; i++) {
                if (neurons[i].val == 0)
                    neurons[i].prev[0].error = 0;
                else
                    neurons[i].prev[0].error = neurons[i].error;
                neurons[i].error = 0;
            }
        }

        public override String ToString () {
            StringBuilder sb = new StringBuilder();
            sb.Append(String.Format("{0}\n{1}", this.GetType().FullName, prob));
            return sb.ToString();
        }
    }
}
