/* 
 * Code to accompany the paper:
 *   Efficient Online Structured Output Learning for Keypoint-Based Object Tracking
 *   Sam Hare, Amir Saffari, Philip H. S. Torr
 *   Computer Vision and Pattern Recognition (CVPR), 2012
 * 
 * Copyright (C) 2012 Sam Hare, Oxford Brookes University, Oxford, UK
 * 
 * This file is part of learnmatch.
 * 
 * learnmatch is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * learnmatch is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with learnmatch.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#include "Video.h"

#include <iostream>

using namespace std;
using namespace cv;

/*Video::Video(const std::string& path, const std::string& format, int nIdxStart, bool write, bool video) :
	m_frameIdx(nIdxStart),
	m_write(write),
	m_bVideo(video)
{
	m_formatString = path + "/" + format;
	if (write)
	{
		cout << "starting video recording to: " << path << endl;
	}

	if (m_bVideo) {
		m_videoCap.open(m_formatString);
	}
}*/

Video::Video(int nIdxStart, bool write, bool video):m_frameIdx(nIdxStart),m_write(write),m_bVideo(video)
{
	if (m_bVideo){
		m_videoCap.open(0);
	}
}


Video::~Video()
{
}


bool Video::WriteFrame(const Mat& frame)
{
	if (!m_write) return false;
	
	char buf[1024];
	sprintf_s(buf, m_formatString.c_str(), m_frameIdx);
	++m_frameIdx;
	
	return imwrite(buf, frame);
}

bool Video::ReadFrame(Mat& rFrame)
{
	if (m_write) return false;
	
	if (m_bVideo) {
		if (m_videoCap.isOpened()) {
			m_videoCap >> rFrame;
		}
	}
	else {
		char buf[1024];
		sprintf_s(buf, m_formatString.c_str(), m_frameIdx);
		++m_frameIdx;

		rFrame = imread(buf, -1);
	}

	return !rFrame.empty();
}
	
bool Video::ReadFrame(std::string& file, cv::Mat& rFrame)
{
	if (m_write) return false;

	char buf[1024];
	sprintf_s(buf, m_formatString.c_str(), file.c_str());
	++m_frameIdx;

	rFrame = imread(buf, -1);

	return !rFrame.empty();
}

bool Video::isVideoCapOpen(){
	if (m_videoCap.isOpened()) return true;
	return false;
}