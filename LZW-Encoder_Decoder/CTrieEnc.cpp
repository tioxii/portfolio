/*!\file CTrieEnc.cpp
 * \brief
 *
 *  Created on: 19.06.2019
 *      Author: tombr
 */

#include "CTrieEnc.hpp"

CTrieEnc::CTrieEnc() {
	for(int i = 0; i < LZW_DICT_SIZE; i++) {  //Dictionary initialisieren
		if(i < 256) {
			m_symbolTable[i].setParent(-1);
			m_symbolTable[i].setSymbol(intToString(i));
		}
		else
			m_symbolTable[i].setParent(-2);
	}
}

CTrieEnc::~CTrieEnc() {
}

string CTrieEnc::getEntry(unsigned int pos) {
	string ret;

		ret = m_symbolTable[pos].getSymbol();

	return ret;
}


vector<unsigned int> CTrieEnc::encode(const string& a) {
	CForwardCounter attemptCounter; 						//counter für rehashing Versuche
	string myString("");
	bool hashAgain(true);
	unsigned int hashPos(0);
	unsigned int parentPos(-1);
	CDoubleHashing& hash = CDoubleHashing::getInstance();   //Hashing instance
	vector<unsigned int> encoded;							//Ausgabevektor
	string::const_iterator it = a.begin();					//iterator



	if(a != "") {											//if-guard, welcher verhindert, dass ein leerer String eingelesen wird
		myString = *it;										//erstes Symbol blind einlesen
		it++;
		parentPos = static_cast<unsigned int> (static_cast<unsigned char> (*myString.begin())); //als ParentPos den ASCII-Wert verwenden, da nur 1 Zeichen
		for(; it != a.end(); it++) {				//iteriert durch den gesamten String

			attemptCounter.setValue(0);				//Zähler der Hash-Versuche beim neuem Zeichen auf 0 setzen
			hashAgain = true;						//hashAgain auf wahr setzen, damit er in die while-Schleife geht
			while(hashAgain) {
				hashPos = hash.hash(parentPos, static_cast<int> (static_cast<char> (*it)), LZW_DICT_SIZE, static_cast<unsigned int> (attemptCounter.getValue())); //Hash-Wert bestimmen

				if(m_symbolTable[hashPos].getParent() == -2) {			//wenn an der Position des Hash-Werts kein Eintrag ist, so wird der aktuelle dort hineingeschrieben
					m_symbolTable[hashPos].setSymbol(myString + *it);
					m_symbolTable[hashPos].setParent(parentPos);
					encoded.push_back(parentPos);
					myString = *it;
					hashAgain = false;									//kei neuer Hash-Versuch benötigt
					parentPos = static_cast<unsigned int> (static_cast<unsigned char> (*myString.begin())); //parentPos neu setzen, da aktuell nur ein Symbol (wieder ASCII-Wert)
				}
				else {
					if(m_symbolTable[hashPos].getSymbol() == myString + *it) {	//falls bereits der selbe eintrag schon an der Stelle vorhanden wird, wird ein neues Zeichen zum String hinzugefügt
						myString += *it;
						parentPos = hashPos;
						hashAgain = false;								//kein neuer Hash-Versuch benötigt, da neues Zeichen zum String hinzugefügt wird
					}
					else {
						if(m_symbolTable[hashPos].getSymbol() != myString + *it) { //schaut ob der Behälter schon belegt ist (Kollision)
							hashAgain = true;				//hier neuer Hash-Versuch benötigt
							attemptCounter.count();			//Zähler für neuen Hash-Versuch um 1 erhöhen
						}
					}

				}
			}
		}
		if(myString.length() == 1) { //letztes Symbol mit geben
			encoded.push_back(static_cast<unsigned int> (static_cast<unsigned char> (*myString.begin())));
		}
		else {
			encoded.push_back(parentPos);
		}

	}
	return encoded; //Vektor zurückgeben
}
