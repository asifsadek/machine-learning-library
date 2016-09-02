using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Machine_Learning.Autoencoder {
    public abstract class Layer {
        public int size;
        public Layer prevLayer;
        public Neuron[] neurons;

        public abstract void BindTo (ref Layer layer);

        public abstract void forwardPropagate ();

        public abstract void backPropagate (double learningRate);
    }
}
