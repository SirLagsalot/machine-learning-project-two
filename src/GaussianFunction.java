public class GaussianFunction implements IActivationFunction {

    @Override
    public double compute(double value) {
        //tanh
        return 0;
    }

    @Override
    public double computeDerivative(double value) {
        // 4/(e^x + e^-x)^2
        return 0;
    }
}
