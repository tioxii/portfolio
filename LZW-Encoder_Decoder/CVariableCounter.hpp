/*!\file CVariableCounter.hpp
 * \brief
 *
 *  Created on: 15.06.2019
 *      Author: tombr
 */

#ifndef COUNTER_CVARIABLECOUNTER_HPP_
#define COUNTER_CVARIABLECOUNTER_HPP_

#include "CCounter.hpp"

/*!
 *
 */
class CVariableCounter: public CCounter {
public:
	CVariableCounter(int step);
	void count();

private:
	int m_step;
};

#endif /* COUNTER_CVARIABLECOUNTER_HPP_ */
