public class GaussianFunction implements IActivationFunction {

    private double sigma = 0.0; // maxDistance/sqrt(2* number of clusters)
    private double mean = 0.0; //RadialBasisNetwork.getMean(); figure this out


    @Override
    public double compute(double value) {
        // 1/(sqrt(2*pi*sigma) * e^(-(current - mean)^2/(2*sigma^2))
        return 1 / (Math.sqrt(2 * Math.PI * sigma) * Math.exp(-Math.pow((value - mean), 2) / (2 * Math.pow(sigma, 2))));
    }

    @Override
    public double computeDerivative(double value) {
        //((x - mu)*e^(-((x - mu)^2)/(2*sigma^2)))/(sqrt(2*pi)*sigma^(5/2))
        return (value - mean) * Math.exp(-(Math.pow((value - mean), 2) / (2 * Math.pow(sigma, 2)))) / (Math.sqrt(2 * Math.PI) * Math.pow(sigma, (5.0 / 2)));
    }

    public double compute(double value, double mean) {
        this.mean = mean;
        return this.compute(value);
    }

    public double computeDerivative(double value, double mean) {
        this.mean = mean;
        return this.computeDerivative(value);
    }

    public void setSigma(double sigma) {
        //maxDistance/sqrt(2* number of clusters)
    }
}
