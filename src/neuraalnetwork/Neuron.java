package neuraalnetwork;

import java.util.List;

public class Neuron {
	
	private double output;
	private List<Connection> outgoingConnections;
	
	public List<Connection> getConnections() {
		return outgoingConnections;
	}
	
	public void addConnection(Neuron outgoingConnection, Weight weight) {
		outgoingConnections.add(new Connection(outgoingConnection, weight));
	}

	public double getOutput() {
		return output;
	}

	public void setOutput(double output) {
		this.output = output;
	}

}
