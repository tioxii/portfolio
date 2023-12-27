/*!\file CCounter.hpp
 * \brief
 *
 *  Created on: 15.06.2019
 *      Author: tombr
 */

#ifndef CCOUNTER_HPP_
#define CCOUNTER_HPP_

/*!
 *
 */
class CCounter {
public:
	CCounter();
	virtual ~CCounter();
	virtual void count() {};
	int getValue()const;
	void setValue(int value);

private:
	int m_value;
};

#endif /* CCOUNTER_HPP_ */
