public class GaussianFunction implements IActivationFunction {

    private static double SIGMA = 0.0;

    @Override
    public double compute(double value) {
        // Value should be the distance between the mean and the inputs for a given neuron
        return 1 / (Math.sqrt(2 * Math.PI * SIGMA) * Math.exp(-Math.pow((value), 2) / (2 * Math.pow(SIGMA, 2))));
    }

    @Override
    public double computeDerivative(double value) {
        // Value should be the distance between the mean and the inputs for a given neuron
        return (value) * Math.exp(-(Math.pow((value), 2) / (2 * Math.pow(SIGMA, 2)))) / (Math.sqrt(2 * Math.PI) * Math.pow(SIGMA, (5.0 / 2)));
    }

    // Sigma is a tunable parameter set using the following heuristic
    public static void setSigma(double max, int clusters) {
        GaussianFunction.SIGMA = max / Math.sqrt(2 * clusters);
    }
}
