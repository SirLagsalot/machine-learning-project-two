public class SigmoidFunction implements IActivationFunction {

    @Override
    public double compute(double value) {
        return 1 / (1 + Math.exp(value * -1));
    }

    @Override
    public double computeDerivative(double value) {
        return value * (1.0 - value);
    }
}
