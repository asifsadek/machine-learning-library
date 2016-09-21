using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Machine_Learning.Neural_Network {

    public class WeightSet {

        // HACKY FOR NOW
        public const double LAMBDA = 0;
        public const double L2_REGULARIZATION = 0.8;
        public const double L1_REGULARIZATION = 1 - L2_REGULARIZATION;
        public const double VELOCITY = 0.0;

        public static Random rand = new Random();

        public double[] val, valVelocity;
        public double bias, biasVelocity;
        public int size;

        public WeightSet (int size) {
            this.size = size;
            this.val = new double[size];
            this.valVelocity = new double[size];
            this.bias = 0;
            this.biasVelocity = 0;

            double range = (1 / Math.Sqrt(size));

            for (int i = 0; i < size; i++)
                val[i] = GetRandomGaussian(range);
        }

        public double evaluate (Neuron[] prev) {
            if (size == 0)
                return 0;
            double ret = bias;
            for (int i = 0; i < size; i++)
                ret += val[i] * prev[i].activated;
            return ret;
        }

        public void updateError (Neuron[] prev, double error) {
            for (int i = 0; i < size; i++)
                prev[i].error += error * val[i];
        }

        public void regularize (double learningRate) {
            for (int i = 0; i < size; i++) {
                double l2Gradient = val[i] * L2_REGULARIZATION;
                double l1Gradient = (val[i] > 0 ? 1 : -1) * L1_REGULARIZATION;
                val[i] -= learningRate * LAMBDA * (l1Gradient + l2Gradient);
            }
        }

        public void update (Neuron[] prev, double error, double learningRate) {
            if (size == 0)
                return;
            for (int i = 0; i < size; i++) {
                double errorGradient = prev[i].activated * error;
                valVelocity[i] = valVelocity[i] * VELOCITY + learningRate * errorGradient;
                val[i] -= valVelocity[i];
            }
            biasVelocity = biasVelocity * VELOCITY + learningRate * error;
            bias -= biasVelocity;
        }

        private double GetRandomGaussian (double stdDev) {
            double u1 = rand.NextDouble();
            double u2 = rand.NextDouble();
            double randStdNormal = (double) (Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
            return stdDev * randStdNormal;
        }
    }
}