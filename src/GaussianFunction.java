public class GaussianFunction implements IActivationFunction {

    @Override
    public double compute(double value) {
        // 1/(sqrt(2*pi*sigma) * e^(-(current - mean)^2/(2*sigma^2))
        return 0;
    }

    @Override
    public double computeDerivative(double value) {
        //((x - mu)*e^(-((x - mu)^2)/(2*sigma^2)))/(sqrt(2*pi)*sigma^(5/2))
        return 0;
    }
}
