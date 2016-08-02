using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Machine_Learning.Neural_Network {

    public class ConvolutionalLayer : Layer2D {

        WeightSet[] sharedWeights;

        public ConvolutionalLayer (int depth, int kernelWidth, int kernelHeight, int type) {
            this.depth = depth;
            this.kernelWidth = kernelWidth;
            this.kernelHeight = kernelHeight;
            this.type = type;
            this.sharedWeights = new WeightSet[depth];
        }

        public ConvolutionalLayer (StreamReader reader, Layer prev) {
            string[] data = reader.ReadLine().Split();
            this.depth = int.Parse(data[0]);
            this.kernelWidth = int.Parse(data[1]);
            this.kernelHeight = int.Parse(data[2]);
            this.type = int.Parse(data[3]);
            this.sharedWeights = new WeightSet[depth];

            BindTo(ref prev);

            for (int i = 0; i < depth; i++) {
                sharedWeights[i].bias = double.Parse(reader.ReadLine());
                for (int j = 0; j < sharedWeights[i].val.GetLength(0); j++) {
                    sharedWeights[i].val[j] = double.Parse(reader.ReadLine());
                }
            }
        }

        public override void BindTo (ref Layer layer) {
            prevLayer = layer;

            Layer2D prev = (Layer2D)prevLayer;
            this.width = prev.width - this.kernelWidth + 1;
            this.height = prev.height - this.kernelHeight + 1;
            this.size = depth * width * height;
            this.neurons = new Neuron[size];

            for (int i = 0; i < depth; i++) {
                sharedWeights[i] = new WeightSet(prev.depth * kernelWidth * kernelHeight);

                for (int j = 0; j < width; j++) {
                    for (int k = 0; k < height; k++) {
                        Neuron[] currPrev = new Neuron[prev.depth * kernelWidth * kernelHeight];

                        for (int x = 0; x < prev.depth; x++)
                            for (int y = 0; y < kernelWidth; y++)
                                for (int z = 0; z < kernelHeight; z++) {
                                    int prevIndex = x * prev.width * prev.height + (j + y) * prev.height + (k + z);
                                    currPrev[x * kernelWidth * kernelHeight + y * kernelHeight + z] = prev.neurons[prevIndex];
                                }

                        neurons[i * width * height + j * height + k] = new Neuron(prev.depth * kernelWidth * kernelHeight, sharedWeights[i], this.type);
                        neurons[i * width * height + j * height + k].link(currPrev);
                    }
                }
            }
        }

        public override void forwardPropagate () {
            for (int i = 0; i < size; i++)
                neurons[i].forwardPropagate();
        }

        public override void backPropagate (double learningRate) {
            for (int i = 0; i < size; i++)
                neurons[i].backPropagateError();

            for (int i = 0; i < size; i++)
                neurons[i].backPropagateWeights(learningRate);
        }

        public override String ToString () {
            StringBuilder sb = new StringBuilder();
            sb.Append(String.Format("{0}\n{1} {2} {3} {4}", this.GetType().FullName, depth, kernelHeight, kernelWidth, type));
            for (int i = 0; i < depth; i++) {
                sb.Append("\n" + sharedWeights[i].bias);
                for (int j = 0; j < sharedWeights[i].val.GetLength(0); j++)
                    sb.Append("\n" + sharedWeights[i].val[j]);
            }
            return sb.ToString();
        }
    }
}
