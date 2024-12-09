#pragma once

namespace nn::interfaces
{
	class CustomBackPropogableInterface {
	public:
		struct Candidate {
			float value;
			void *id;
		};
		virtual const std::vector<Candidate> &RetriveCandidates() = 0;
		virtual void SelectBestCandidate(void *id, float error) = 0;
		virtual bool IsCustomBackPropAvailable() = 0;
	};
}