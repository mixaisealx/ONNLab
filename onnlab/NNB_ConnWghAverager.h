#pragma once
#include "BasicLayerI.h"

#include <vector>

namespace nn
{
	// A set of connections forming a virtual hypergraph is stored here. The peculiarity of this hypergraph is that the compounds included in it have the same weight.
	class NNB_ConnWghAverager {
		std::vector<interfaces::ConnectionBasicInterface *> connections;

	public:
		NNB_ConnWghAverager() {};

		NNB_ConnWghAverager(std::initializer_list<interfaces::ConnectionBasicInterface *> connections):connections(connections) {
		};

		const std::vector<interfaces::ConnectionBasicInterface *> &Connections() {
			return connections;
		}

		void AddConnetion(interfaces::ConnectionBasicInterface *connection) {
			connections.push_back(connection);
		}

		void DoWeightsProcessing() {
			float summ = 0;
			for (auto conn : connections) {
				summ += conn->Weight();
			}
			summ /= connections.size();
			for (auto conn : connections) {
				conn->Weight(summ);
			}
		}
	};
}
