using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Machine_Learning {
    public abstract class Kernel {
        public int size;

        public abstract double[] getFeatures (double[] data);
    }

    public class LinearKernel : Kernel {
        public LinearKernel (int size) {
            this.size = size;
        }

        public override double[] getFeatures (double[] data) {
            return data;
        }
    }

    public class GaussianKernel : Kernel {
        double gamma; // gamma = 1 / (2 * sigma ^ 2) where sigma is the kernal width paramter
        double[][] val;

        public GaussianKernel (double gamma, double[][] val) {
            this.gamma = gamma;
            this.val = val;
            this.size = val.GetLength(0);
        }

        public override double[] getFeatures (double[] data) {
            double[] ret = new double[size];

            for (int i = 0; i < size; i++)
                ret[i] = getKernelValue(data, val[i]);

            return ret;
        }

        private double getKernelValue (double[] x, double[] y) {
            double ret = 0;
            Debug.Assert(x.GetLength(0) == x.GetLength(0));
            for (int i = 0; i < x.GetLength(0); i++)
                ret += (x[i] - y[i]) * (x[i] - y[i]);
            return Math.Exp(-gamma * ret);
        }
    }
}
