public class HyperbolicFunction implements IActivationFunction {

    @Override
    public double compute(double value) {
        return Math.tanh(value);
    }

    @Override
    public double computeDerivative(double value) {
        return 1 - Math.pow(value, 2);
    }
}
