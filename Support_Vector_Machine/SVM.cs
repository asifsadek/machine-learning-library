using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Machine_Learning.Support_Vector_Machine {

    public class SVM {

        public Kernel kernel;
        public int featureSize;
        public double[] weights;
        public double bias;
        public static Random rand = new Random();

        public SVM (Kernel kernel) {
            this.kernel = kernel;
            this.featureSize = kernel.size;
            this.weights = new double[featureSize];

            for (int i = 0; i < featureSize; i++)
                weights[i] = GetRandomGaussian(1);
            bias = 0;
        }

        // answer should be either -1 or 1
        public double train (double[] data, double answer, double learningRate) {
            double res = bias;
            double[] features = kernel.getFeatures(data);
            for (int i = 0; i < featureSize; i++)
                res += features[i] * weights[i];

            double cost = Math.Max(0, 1 - answer * res);
            double error = cost == 0 ? 0 : -answer;
            for (int i = 0; i < featureSize; i++)
                weights[i] -= error * features[i] * learningRate;
            bias -= error * learningRate;

            return cost;
        }

        // return -1 or 1
        public int classify (double[] data) {
            double ret = bias;
            double[] features = kernel.getFeatures(data);
            for (int i = 0; i < featureSize; i++)
                ret += features[i] * weights[i];

            return ret >= 0 ? 1 : -1;
        }

        private double GetRandomGaussian (double stdDev) {
            double u1 = rand.NextDouble();
            double u2 = rand.NextDouble();
            double randStdNormal = (double)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2));
            return stdDev * randStdNormal;
        }
    }
}
