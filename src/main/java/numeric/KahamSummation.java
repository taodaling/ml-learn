package numeric;

import java.math.BigDecimal;


/**
 * O(1) error for real number summation
 */
public class KahamSummation {
    private double error;
    private double sum;

    public void reset() {
        error = 0;
        sum = 0;
    }

    public double sum() {
        return sum;
    }

    public void add(double x) {
        x = x - error;
        double t = sum + x;
        error = (t - sum) - x;
        sum = t;
    }

    public void subtract(double x) {
        add(-x);
    }


    @Override
    public String toString() {
        return new BigDecimal(sum).toString();
    }
}
