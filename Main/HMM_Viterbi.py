#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np

class HMMViterbi:
    """Hidden Markov Model (HMM) Viterbi class for sequence tagging."""

    def __init__(self, states=[], observations=[], initial_probabilities={}, transition_probabilities={},
                 emission_probabilities={}, transition_matrix=[]):
        self.states = states
        self.observations = observations
        self.initial_probabilities = initial_probabilities
        self.transition_probabilities = transition_probabilities
        self.emission_probabilities = emission_probabilities
        self.transition_matrix = transition_matrix
        self.forward_values = []
        self.backward_values = []
        self.prob_forward = []
        self.posterior_values = []

    def set_parameters(self, encoded_text_list):
        self.states = list(range(max(encoded_text_list)))
        self.observations = list(range(max(encoded_text_list)))
        self.initial_probabilities = self.calculate_initial_probabilities(encoded_text_list)
        self.transition_matrix = self.calculate_transition_matrix(encoded_text_list)
        self.transition_probabilities = self.convert_to_dict(self.transition_matrix)
        self.emission_probabilities = self.convert_to_dict(self.transition_matrix.transpose(1, 0))

    def calculate_initial_probabilities(self, encoded_text_list):
        initial_probabilities = {}
        unique_states = sorted(set(encoded_text_list))
        transition_probabilities = []
        length = len(encoded_text_list)

        for x in unique_states:
            transition_probabilities.append(len(np.where(encoded_text_list == x)[0]) / length)
        probabilities_array = np.array(transition_probabilities)

        for i, prob in zip(list(range(len(probabilities_array)), probabilities_array):
            initial_probabilities[i] = prob
        return initial_probabilities

    def calculate_transition_matrix(self, encoded_text_list):
        num_states = 1 + max(encoded_text_list)
        transition_matrix = np.zeros(shape=(num_states, num_states))

        for i in range(0, num_states):
            state_indices = list(np.where(encoded_text_list == i)[0])
            for index in state_indices:
                if (index + 1 != len(encoded_text_list)):
                    transition_matrix[i, encoded_text_list[index + 1]] += 1 / len(state_indices)

        transition_matrix[transition_matrix != 0] = transition_matrix[transition_matrix != 0] + (1 / num_states)
        transition_matrix[transition_matrix == 0] = 1 / num_states

        return transition_matrix

    def convert_to_dict(self, matrix):
        dictionary = {}
        for i, row in zip(list(range(np.size(matrix, axis=0)), matrix):
            temp = {}
            for j, val in zip(list(range(np.size(row, axis=0)), row):
                temp[j] = val
            dictionary[i] = temp
        return dictionary

    def viterbi_algorithm(self, transition_matrix, emission_matrix, observations):
        num_observations = len(observations)
        num_states = transition_matrix.shape[0]
        log_probs = np.zeros(num_states)
        paths = np.zeros((num_states, num_observations + 1))
        paths[:, 0] = np.arange(num_states)

        for obs_ind, obs_val in enumerate(observations):
            for state_ind in range(num_states):
                value = 0
                if obs_val < np.size(emission_matrix, 1):
                    value = np.log(emission_matrix[state_ind, obs_val])
                temp_probs = log_probs + value + np.log(transition_matrix[:, state_ind])
                best_temp_ind = np.argmax(temp_probs)
                paths[state_ind, :] = paths[best_temp_ind, :]
                paths[state_ind, (obs_ind + 1)] = state_ind
                log_probs[state_ind] = temp_probs[best_temp_ind]

        best_path_ind = np.argmax(log_probs)

        return (paths[best_path_ind], log_probs[best_path_ind])

    def calculate_posterior(self):
        posterior = []
        for i in range(len(self.observations)):
            posterior.append({
                st: self.forward_values[i][st] * self.backward_values[i][st] / self.prob_forward
                for st in self.states
            })

        return posterior

    def forward_algorithm(self, end_state):
        forward_values = []
        f_prev = {}
        for i, observation_i in enumerate(self.observations):
            f_curr = {}
            for st in self.states:
                if i == 0:
                    prev_f_sum = self.initial_probabilities[st]
                else:
                    prev_f_sum = sum(f_prev[k] * self.transition_probabilities[k][st] for k in self.states)

                f_curr[st] = self.emission_probabilities[st][observation_i] * prev_f_sum

            forward_values.append(f_curr)
            f_prev = f_curr

        prob_forward = sum(f_curr[k] * self.transition_probabilities[k][end_state] for k in self.states)

        return forward_values, prob_forward

    def backward_algorithm(self, end_state='PERIOD'):
        backward_values = []
        b_prev = {}
        for i, observation_i_plus in enumerate(reversed(self.observations[1:] + [None, ])):
            b_curr = {}
            for st in self.states:
                if i == 0:
                    b_curr[st] = self.transition_probabilities[st][end_state]
                else:
                    b_curr[st] = sum(
                        self.transition_probabilities[st][l] * self.emission_probabilities[l][observation_i_plus] * b_prev[l]
                        for l in self.states
                    )

            backward_values.insert(0, b_curr)
            b_prev = b_curr

        p_bkw = sum(
            self.initial_probabilities[l] * self.emission_probabilities[l][self.observations[0]] * b_curr[l]
            for l in self.states
        )

        return backward_value

