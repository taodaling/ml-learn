package utils;

public class MathUtils {
    public static final double INF = (double) 1e100;

    public static double log(double x) {
        if (x == 0) {
            return -INF;
        }
        return Math.log(x);
    }
}
