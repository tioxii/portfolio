/*!\file CVariableCounter.cpp
 * \brief
 *
 *  Created on: 15.06.2019
 *      Author: tombr
 */

#include "CVariableCounter.hpp"

CVariableCounter::CVariableCounter(int step): CCounter(), m_step(step) {

}

void CVariableCounter::count() {
	setValue(getValue() + m_step);
}
