using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Machine_Learning {
    public class DropoutLayer : Layer {
        static Random rand = new Random();

        // probability to keep neuron
        public double prob;

        public DropoutLayer (double prob) {
            this.prob = prob;
        }

        public DropoutLayer (StreamReader reader, Layer prev) {
            string[] data = reader.ReadLine().Split();
            this.size = int.Parse(data[0]);
            this.prob = double.Parse(data[1]);

            BindTo(ref prev);
        }

        public override void BindTo (ref Layer layer) {
            prevLayer = layer;

            this.neurons = new Neuron[prevLayer.size];

            for (int i = 0; i < size; i++) {
                neurons[i] = new Neuron(1, this.type);
                neurons[i].link(new Neuron[]{prevLayer.neurons[i]});
            }
        }

        public override void forwardPropagate () {
            for (int i = 0; i < size; i++)
                if (rand.NextDouble() < prob)
                    neurons[i].val = neurons[i].activated = neurons[i].prev[0].activated;
        }

        public void forwardPropagateTest () {
            for (int i = 0; i < size; i++)
                neurons[i].val = neurons[i].activated = neurons[i].prev[0].activated * prob;
        }

        public override void backPropagate (double learningRate) {
            for (int i = 0; i < size; i++) {
                neurons[i].prev[0].error = neurons[i].error;
                neurons[i].error = 0;
            }
        }

        public override String ToString () {
            StringBuilder sb = new StringBuilder();
            sb.Append(String.Format("{0}\n{1} {2}", "Machine_Learning.OutputLayer", size, prob));
            return sb.ToString();
        }
    }
}
