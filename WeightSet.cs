using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Machine_Learning {

    public class WeightSet {

        // HACKY FOR NOW
        public const double LAMBDA = 0.0001F;

        public static Random rand = new Random();

        public double[] val;
        public double bias;
        public int size;

        public WeightSet (int size) {
            this.size = size;
            this.val = new double[size];
            this.bias = 0;

            double range = (1 / Math.Sqrt(size));

            for (int i = 0; i < size; i++)
                val[i] = GetRandomGaussian(range);
        }

        public double evaluate (Neuron[] prev) {
            double ret = bias;
            for (int i = 0; i < size; i++)
                ret += val[i] * prev[i].activated;
            return ret;
        }

        public void updateError (Neuron[] prev, double error) {
            for (int i = 0; i < size; i++)
                prev[i].error += error * val[i];
        }

        public void update (Neuron[] prev, double error, double learningRate) {
            for (int i = 0; i < size; i++)
                val[i] -= learningRate * (prev[i].activated * error + LAMBDA * val[i]);
            bias -= error * learningRate;
        }

        private double GetRandomGaussian (double stdDev) {
            double u1 = rand.NextDouble();
            double u2 = rand.NextDouble();
            double randStdNormal = (double) (Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
            return stdDev * randStdNormal;
        }
    }
}
