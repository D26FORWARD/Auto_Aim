/****************************************************************************
 *  Copyright (C) 2019 RoboMaster.
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of 
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program. If not, see <http://www.gnu.org/licenses/>.
 ***************************************************************************/

#include "serial_device.h"
#include <iostream>

//namespace robomaster {
SerialDevice::SerialDevice(std::string port_name,
						   int baudrate) :
	port_name_(port_name),
	baudrate_(baudrate),
	data_bits_(8),
	parity_bits_('N'),
	stop_bits_(1)
{
}

SerialDevice::~SerialDevice()
{
	CloseDevice();
}

// 初始化
bool SerialDevice::Init()
{

	std::cout << "Attempting to open device " << port_name_
		<< " with baudrate " << baudrate_ << std::endl;
	if (port_name_.c_str() == nullptr)
	{
		port_name_ = "/dev/ttyUSB0";
	}
	if (OpenDevice() && ConfigDevice())
	{
		FD_ZERO(&serial_fd_set_);
		FD_SET(serial_fd_, &serial_fd_set_);
		std::cout << "Serial started successfully." << std::endl;
		return true;
	}
	else
	{
		std::cerr << "Failed to start serial " << port_name_ << std::endl;
		CloseDevice();
		return false;
	}
}

// 打开设备
bool SerialDevice::OpenDevice()
{

#ifdef __arm__
	serial_fd_ = open(port_name_.c_str(), O_RDWR | O_NONBLOCK);
#elif __x86_64__
	serial_fd_ = open(port_name_.c_str(), O_RDWR | O_NOCTTY);
#else
	serial_fd_ = open(port_name_.c_str(), O_RDWR | O_NOCTTY);
#endif

	if (serial_fd_ < 0)
	{
		std::cerr << "Cannot open device "
			<< port_name_ << std::endl;
		return false;
	}

	return true;
}

// 关闭通讯协议接口
bool SerialDevice::CloseDevice()
{
	close(serial_fd_);
	serial_fd_ = -1;
	return true;
}

// 配置设备
bool SerialDevice::ConfigDevice()
{
	int st_baud[] = { B4800, B9600, B19200, B38400,
					 B57600, B115200, B230400, B921600 };
	int std_rate[] = { 4800, 9600, 19200, 38400, 57600, 115200,
					  230400, 921600, 1000000, 1152000, 3000000 };
	int i, j;
	/* save current port parameter */
	if (tcgetattr(serial_fd_, &old_termios_) != 0)
	{
		std::cerr << "fail to save current port" << std::endl;
		return false;
	}
	memset(&new_termios_, 0, sizeof(new_termios_));

	/* config the size of char */
	new_termios_.c_cflag |= CLOCAL | CREAD;
	new_termios_.c_cflag &= ~CSIZE;

	/* config data bit */
	switch (data_bits_)
	{
	case 7:
		new_termios_.c_cflag |= CS7;
		break;
	case 8:
		new_termios_.c_cflag |= CS8;
		break;
	default:
		new_termios_.c_cflag |= CS8;
		break; //8N1 default config
	}
	/* config the parity bit */
	switch (parity_bits_)
	{
/* odd */
	case 'O':
	case 'o':
		new_termios_.c_cflag |= PARENB;
		new_termios_.c_cflag |= PARODD;
		break;
		/* even */
	case 'E':
	case 'e':
		new_termios_.c_cflag |= PARENB;
		new_termios_.c_cflag &= ~PARODD;
		break;
		/* none */
	case 'N':
	case 'n':
		new_termios_.c_cflag &= ~PARENB;
		break;
	default:
		new_termios_.c_cflag &= ~PARENB;
		break; //8N1 default config
	}
	/* config baudrate */
	j = sizeof(std_rate) / 4;
	for (i = 0; i < j; ++i)
	{
		if (std_rate[i] == baudrate_)
		{
/* set standard baudrate */
			cfsetispeed(&new_termios_, st_baud[i]);
			cfsetospeed(&new_termios_, st_baud[i]);
			break;
		}
	}
	/* config stop bit */
	if (stop_bits_ == 1)
		new_termios_.c_cflag &= ~CSTOPB;
	else if (stop_bits_ == 2)
		new_termios_.c_cflag |= CSTOPB;
	else
		new_termios_.c_cflag &= ~CSTOPB; //8N1 default config

/* config waiting time & min number of char */
	new_termios_.c_cc[VTIME] = 1;
	new_termios_.c_cc[VMIN] = 18;

	/* using the raw data mode */
	new_termios_.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
	new_termios_.c_oflag &= ~OPOST;

	/* flush the hardware fifo */
	tcflush(serial_fd_, TCIFLUSH);

	/* activite the configuration */
	if ((tcsetattr(serial_fd_, TCSANOW, &new_termios_)) != 0)
	{
		std::cerr << "failed to activate serial configuration" << std::endl;
		return false;
	}
	return true;

}

int SerialDevice::Read(uint8_t* buf, int len)
{
	int ret = -1;

	if (NULL == buf)
	{
		return -1;
	}
	else
	{
		ret = read(serial_fd_, buf, len);
		// std::cout<<"Read once length: "<<ret<<std::endl;

		while (ret == 0)
		{
			std::cerr << "Connection closed, try to reconnect." << std::endl;
			while (!Init())
			{
				usleep(500000);//check every 500 ms
			}
			std::cout << "Reconnect Success." << std::endl;
			ret = read(serial_fd_, buf, len);
		}
		return ret;
	}
}

int SerialDevice::ReadUntil2(uint8_t* buf, uint8_t end1, uint8_t end2, uint8_t max_len)
{
	int ret = -1;
	// int flag = 0; // flag 变量在这里是冗余的，因为你直接使用了 return
	uint8_t* p = buf;
	uint8_t i = 0;

	if (NULL == buf)
	{
		return -1; // 错误：缓冲区为空
	}
	else
	{
		while (true)
		{ // 改为无限循环，因为你已经在内部处理了所有退出条件
			ret = read(serial_fd_, p, 1);

			while (ret == 0)
			{ // 如果连接关闭，尝试重连
				std::cerr << "Connection closed, try to reconnect." << std::endl;
				while (!Init())
				{
					usleep(500000); // 检查每 500 ms
				}
				std::cout << "Reconnect Success." << std::endl;
				ret = read(serial_fd_, p, 1);
			}

			if (*p == end1)
			{
				p++;
				// 确保有足够的空间读取下一个字节，并且 read 返回有效数据
				// 这里的 read 应该有错误处理，否则 p 可能指向非法内存
				int next_byte_ret = read(serial_fd_, p, 1);
				if (next_byte_ret <= 0)
				{
				// 如果读取下一个字节失败（例如连接再次关闭或 EOF），
				// 你需要决定如何处理。这里我假设可以返回一个错误码或重试。
				// 为了简化，暂时直接返回一个错误
					return -2; // 例如，表示读取第二个结束字节失败
				}

				if (*p == end2)
				{
   				// 找到两个结束字节
					tcflush(serial_fd_, TCIFLUSH);
					return 1; // 成功找到并读取到结束序列
				}
			}

			// 检查是否超出最大长度，这里先增加 i 再检查，确保 p 指向下一个可写入位置
			i++;
			p++; // 移动指针到下一个位置，准备读取下一个字节

			if (i >= max_len)
			{
			// 达到最大读取长度，但未找到结束序列
				tcflush(serial_fd_, TCIFLUSH);
				return 0; // 达到最大长度，但未找到结束序列
			}
			// 注意：如果 *p == end1 不成立，或者 *p == end2 不成立，循环会继续
		}
	}
	// 理论上，根据上述逻辑，代码不会到达这里。
	// 但为了满足编译器，添加一个默认返回，表示一个“不可能发生”的错误或异常终止
	// 这里的返回值通常表示一个非常规的、不应该发生的情况
	return -3; // 示例：表示未知错误或逻辑漏洞
}

// 发送数据函数
int SerialDevice::Write(const uint8_t* buf, int len)
{
	return write(serial_fd_, buf, len);
}
//}
