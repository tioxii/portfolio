/*!\file XOutOfBounds.cpp
 * \brief
 *
 *  Created on: 16.06.2019
 *      Author: tombr
 */

#include "XOutOfBounds.hpp"

XOutOfBounds::XOutOfBounds(const char* msg): m_msg(static_cast<std::string> (msg)) { //Initialisierung von m_msg mit static cast

}

XOutOfBounds::~XOutOfBounds() {

}

const char* XOutOfBounds::what() const noexcept{ //Keine Ausnahmen erlaubt in expection Klasse
	return m_msg.c_str();
}
