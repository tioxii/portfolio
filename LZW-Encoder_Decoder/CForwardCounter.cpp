/*!\file CForwardCounter.cpp
 * \brief
 *
 *  Created on: 15.06.2019
 *      Author: tombr
 */

#include "CForwardCounter.hpp"

CForwardCounter::CForwardCounter(): CCounter() {


}

void CForwardCounter::count() {
	setValue(getValue()+1);
}
