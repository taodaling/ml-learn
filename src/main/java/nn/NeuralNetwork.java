package nn;

import algo.Hypothesis;
import numeric.Vector;

import java.util.Arrays;

public class NeuralNetwork implements Hypothesis<double[]>, Cloneable {
    double[][][] theta;

    public NeuralNetwork(double[][][] theta) {
        this.theta = theta;
    }

    public double[] apply(double[] xs) {
        return NeuralNetworkUtils.forwardPropagation(xs, theta).a[theta.length];
    }

    public double[][][] getTheta() {
        return theta;
    }

    @Override
    public NeuralNetwork clone() {
        try {
            NeuralNetwork ans = (NeuralNetwork) super.clone();
            ans.theta = Vector.deepClone(ans.theta);
            return ans;
        } catch (CloneNotSupportedException e) {
            throw new UnknownError();
        }
    }

    @Override
    public String toString() {
        return "Theta: \n" + Arrays.deepToString(theta);
    }
}
