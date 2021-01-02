package algo;

public class GradientLearnRateController {
    private double learnRate;
    private double cost;
    private static int mask = (1 << 3) - 1;
    private int state;
    private int ascend;
    private int round;
    private int roundThreshold;

    public double getCost() {
        return cost;
    }

    public GradientLearnRateController(double init, double cost, int roundThreshold) {
        this.learnRate = init;
        this.cost = cost;
        this.roundThreshold = roundThreshold;
    }

    public boolean hasNext() {
        return round < roundThreshold && learnRate >= 1e-15 &&
                !Double.isNaN(cost) && !Double.isInfinite(cost);
    }

    public int getRound() {
        return round;
    }

    public double getLearnRate() {
        return learnRate;
    }

    public void next(double cost) {
        round++;
        state = (state << 1) & mask;
        if (cost >= this.cost) {
            state |= 1;
            ascend = 0;
        } else {
            ascend++;
        }
        if (Integer.bitCount(state) * 2 > Math.min(round, 3)) {
            state = 0;
            learnRate *= 0.5;
        }
        if (ascend >= 3) {
            ascend = 0;
            learnRate *= 1.1;
        }
        this.cost = cost;
    }

    @Override
    public String toString() {
        return String.format("%d-th round, learn rate is %.12f, cost is %.12f", round, learnRate, cost);
    }
}
