/*!\file XOutOfBounds.hpp
 * \brief
 *
 *  Created on: 16.06.2019
 *      Author: tombr
 */

#ifndef ARRAY_XOUTOFBOUNDS_HPP_
#define ARRAY_XOUTOFBOUNDS_HPP_

#include <exception>
#include <string>

/*!
 *
 */
class XOutOfBounds: public std::exception {
public:
	XOutOfBounds(const char* msg);
	virtual ~XOutOfBounds() throw();
	const char* what() const noexcept;

private:
	std::string m_msg;
};

#endif /* ARRAY_XOUTOFBOUNDS_HPP_ */
