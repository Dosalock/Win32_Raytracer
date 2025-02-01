/*******************************************************************************
 *
 *  @file      helper.h
 *  @brief     Helper functions for color calculations etc.
 *  @author    Johan Karlsson - github.com/dosalock
 *  @date      02.02.2024
 *  @copyright Copyright © [2025] [Johan Karlsson]
 *
 ******************************************************************************/


#ifndef __HELPER_H__
#define __HELPER_H__


/*---------------------Includes---------------------------*/
#include <Windows.h>
#include <algorithm>
#include <cstdint>

struct WideColor
{
    uint32_t red;
    uint32_t green;
    uint32_t blue;

    WideColor operator+ ( const WideColor &other ) const
    {
        return WideColor( red + other.red,
                          green + other.green,
                          blue + other.blue );
    }

    WideColor operator- ( const WideColor &other ) const
    {
        return WideColor( red - other.red,
                          green - other.green,
                          blue - other.blue );
    }
};

/**
 * @brief Clamps color channel between 0 and 255
 *
 * @param color - Color channel
 *
 * @return Returns a uint16_t color channel thats been clamped
 */
inline COLORREF ClampColor ( WideColor color )
{
    uint8_t red   = std::clamp( color.red, 0u, 255u );
    uint8_t green = std::clamp( color.green, 0u, 255u );
    uint8_t blue  = std::clamp( color.blue, 0u, 255u );
    return RGB( red, green, blue );
}

/**
 * @brief Applies multiplier to color, channel wise
 *
 * @param[in] color - Color, COLOREF
 * @param[in] multiplier - Multiplier
 *
 * @return Color product of color and multiplier
 */
inline WideColor ApplyMultiplierToColor ( _In_ const WideColor &color,
                                          _In_ const float &multiplier )
{
    uint32_t red   = static_cast<uint32_t>( color.red * multiplier );
    uint32_t green = static_cast<uint32_t>( color.green * multiplier );
    uint32_t blue  = static_cast<uint32_t>( color.blue * multiplier );

    return { red, green, blue };
}

/**
 * @brief Calculates final color of point, taking reflectiveness and reflection
 * modifiers into account
 *
 * @param lit_color - Base color after light calculations
 * @param reflected_color - Color from inverted ray of reflection
 * @param reflectiveness - Reflectiveness of the sphere where the point exists
 *
 * @return
 */
inline WideColor CalculateFinalColor ( _In_ const WideColor &lit_color,
                                       _In_ const WideColor &reflected_color,
                                       _In_ const float &reflectiveness )
{
    float lit_multiplier = ( 1.0f - reflectiveness );

    /* Calculate components */
    WideColor multiplied_lit_color =
        ApplyMultiplierToColor( lit_color, lit_multiplier );
    WideColor multiplied_reflected_color =
        ApplyMultiplierToColor( reflected_color, reflectiveness );

    return multiplied_lit_color + multiplied_reflected_color;
}

/**
 * @brief Sets pixel at offset to color
 *
 * @param[in,out] p_lpv_bits - Pointer to pixel buffer
 * @param[in]     offset - Offset into pixel buffer for wanted pixel
 * @param[in]     color - Wanted color of pixel
 */
inline void SetPixelToColor ( _Inout_ BYTE **p_lpv_bits,
                              _In_ const uint32_t &offset,
                              _In_ const COLORREF &color )
{
    ( *p_lpv_bits )[offset + 0] = static_cast<uint32_t>( GetBValue( color ) );
    ( *p_lpv_bits )[offset + 1] = static_cast<uint32_t>( GetGValue( color ) );
    ( *p_lpv_bits )[offset + 2] = static_cast<uint32_t>( GetRValue( color ) );
    ( *p_lpv_bits )[offset + 3] = 255;
};

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
inline bool IsInBounds ( const ValueType &value,
                         const LowType &low,
                         const HighType &high )
{
    // Comparison logic will go here
    return !( value < low ) && ( value < high );
}

#endif // !__HELPER_H__
