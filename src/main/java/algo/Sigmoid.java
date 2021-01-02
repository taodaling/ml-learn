package algo;

public class Sigmoid {
    public static double apply(double z) {
        return 1 / (1 + Math.exp(-z));
    }
}
