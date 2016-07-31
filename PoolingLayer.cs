using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Machine_Learning {

    public abstract class PoolingLayer : Layer2D {

        public PoolingLayer (int kernelWidth, int kernelHeight) {
            this.kernelWidth = kernelWidth;
            this.kernelHeight = kernelHeight;
        }

        public PoolingLayer () { }
    }
}
