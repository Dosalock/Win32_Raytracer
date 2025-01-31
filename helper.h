#pragma once
/*---------------------Includes---------------------------*/
#include <Windows.h>
#include <algorithm>
#include <functional>
#include <limits>
/**
 * @brief Applies multiplier to color, channel wise
 *
 * @param[in] color - Color, COLOREF
 * @param[in] multiplier - Multiplier
 *
 * @return Color product of color and multiplier
 */
COLORREF ApplyMultiplierToColor ( uint32_t &red, uint32_t &green, uint32_t &blue, _In_ float &multiplier )
{
    uint32_t uint8_max = static_cast<uint32_t>( UINT8_MAX );
	uint32_t uint8_min = 0;
	uint32_t r = red * multiplier;
	uint32_t g = green * multiplier;
	uint32_t b = blue * multiplier;
    return RGB( std::clamp( r, uint8_min, uint8_max ), 
				std::clamp( g, uint8_min, uint8_max ), 
				std::clamp( b, uint8_min, uint8_max ) );
}

/**
 * @brief Calculates final color of point, taking reflectiveness and reflection
 * modifiers into account
 *
 * @param[in] base_color - Color after light calculation
 * @param[in] reflected_color - Color from inverted ray of reflection
 * @param[in] reflectiveness - Reflectiveness of the sphere where the point
 * exists
 *
 * @return Final color clamped beteen 0 and 255
 */
COLORREF CalculateFinalColor ( COLORREF &base_color,
                               COLORREF &reflected_color,
                               float &reflectiveness )
{
    float base_color_reflectiveness_multiplier = 1 - reflectiveness;

     /*
    COLORREF reflected_color_result =
	 ApplyMultiplierToColor( reflected_color, reflectiveness );
    COLORREF base_color_result =
        ApplyMultiplierToColor( base_color,
                                base_color_reflectiveness_multiplier );*/
    uint32_t uint8_max = static_cast<uint32_t>( UINT8_MAX );
    uint32_t uint8_min = 0;
	uint32_t r = static_cast<uint32_t>(GetRValue( reflected_color )) * reflectiveness; 
	uint32_t g = static_cast<uint32_t>(GetGValue( reflected_color )) * reflectiveness; 
	uint32_t b = static_cast<uint32_t>(GetBValue( reflected_color )) * reflectiveness;
	r = std::clamp(r,uint8_min,uint8_max);
	g = std::clamp(g,uint8_min,uint8_max);
	b = std::clamp(b,uint8_min,uint8_max);
    uint32_t lit_r = static_cast<uint32_t>(GetRValue( base_color )) *  (1 - reflectiveness);
	uint32_t lit_g = static_cast<uint32_t>(GetGValue( base_color )) *  (1 - reflectiveness);
	uint32_t lit_b = static_cast<uint32_t>(GetBValue( base_color )) *  (1 - reflectiveness);
    return 	RGB(  
				std::clamp( lit_r + r,
                            static_cast<uint32_t>( 0 ),
                            static_cast<uint32_t>( 255 ) ),
                std::clamp( lit_g + g,
                            static_cast<uint32_t>( 0 ),
                            static_cast<uint32_t>( 255 ) ),
                std::clamp( lit_b + b,
                            static_cast<uint32_t>( 0 ),
                            static_cast<uint32_t>( 255 ) ) );
    //return ClampColor( ColorAddition(base_color_result, reflected_color_result ));
}

/**
 * @brief Sets pixel at offset to color
 *
 * @param[in,out] p_lpv_bits - Pointer to pixel buffer1
 * @param[in]     offset - Offset into pixel buffer for wanted pixel
 * @param[in]     color - Wanted color of pixel
 */
void SetPixelToColor ( _Inout_ BYTE **p_lpv_bits,
                       _In_ uint32_t &offset,
                       _In_ COLORREF &color )
{
    ( *p_lpv_bits )[offset + 0] = static_cast<uint16_t>( GetBValue( color ) );
    ( *p_lpv_bits )[offset + 1] = static_cast<uint16_t>( GetGValue( color ) );
    ( *p_lpv_bits )[offset + 2] = static_cast<uint16_t>( GetRValue( color ) );
    ( *p_lpv_bits )[offset + 3] = 255;
}

/*------------Template Declarations---------------*/
template<typename T>
concept Scalar = std::is_scalar_v<T>;

/**
 * @brief Returns true if a value is between, low and high
 *
 * @tparam T - Any scalar, std::is_scalar_v<T> == true
 *
 * @param value - value to check if its in bounds
 * @param low - lower bound
 * @param high - upper bound
 *
 * @return
 */
template<Scalar ValueType, Scalar LowType, Scalar HighType>
bool IsInBounds ( const ValueType &value,
                  const LowType &low,
                  const HighType &high )
{
    // Comparison logic will go here
    return !( value < low ) && ( value < high );
}
