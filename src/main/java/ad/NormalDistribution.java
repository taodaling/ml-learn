package ad;

public class NormalDistribution {
    private static final double fix = 1.0 / Math.sqrt(2 * Math.PI);
    private double sigma;
    private double mu;


    public NormalDistribution(double mu, double sigma) {
        this.sigma = sigma;
        this.mu = mu;
    }

    public double apply(double x) {
        double prob = fix / sigma * Math.exp(-(x - mu) * (x - mu) / (2 * sigma * sigma));
        assert prob >= 0 && prob <= 1 + 1e-8;
        return prob;
    }
}
