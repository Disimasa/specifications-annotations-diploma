"""
Модуль для фильтрации неинформативных сегментов текста.
"""
import re
from typing import List


class SegmentFilter:
    """
    Фильтр для удаления неинформативных сегментов из текста.
    """
    
    # Паттерны неинформативных фраз (точные совпадения, регистронезависимо)
    NON_INFORMATIVE_PHRASES = {
        "проект",
        "утверждаю",
        "техническое задание",
        "согласовано",
        "приложение",
        "содержание",
        "раздел",
        "задачи:",
        "задача:",
        "цель:",
        "цель",
    }
    
    # Паттерны заголовков разделов (регулярные выражения)
    SECTION_HEADER_PATTERNS = [
        r"^РАЗДЕЛ\s+\d+\.?\s*$",  # "РАЗДЕЛ 1.", "РАЗДЕЛ 2"
        r"^Раздел\s+\d+\.?\s*$",  # "Раздел 1.", "Раздел 2"
        r"^РАЗДЕЛ\s+\d+\.?\s+[А-ЯЁ\s]+$",  # "РАЗДЕЛ 1. НАИМЕНОВАНИЕ" (только заголовок)
    ]
    
    # Паттерны дат
    DATE_PATTERNS = [
        r"^\d{1,2}\.\d{1,2}\.\d{2,4}$",  # "31.10.2023", "01.01.23"
        r"^\d{1,2}\.\d{1,2}\.\d{2,4}\s*$",  # с пробелами
    ]
    
    # Паттерны места и даты
    LOCATION_DATE_PATTERNS = [
        r"^[А-ЯЁ][а-яё\s]+,\s*г\.\s*[А-ЯЁ][а-яё]+\s+\d{4}$",  # "Московская область, г. Подольск 2023"
        r"^[А-ЯЁ][а-яё]+\s+\d{4}$",  # "Москва 2024"
    ]
    
    # Минимальная длина информативного сегмента (в символах)
    MIN_INFORMATIVE_LENGTH = 15
    
    def __init__(
        self,
        min_length: int = MIN_INFORMATIVE_LENGTH,
        filter_headers: bool = True,
        filter_dates: bool = True,
        filter_phrases: bool = True
    ):
        """
        Инициализация фильтра.
        
        Args:
            min_length: Минимальная длина сегмента для сохранения
            filter_headers: Фильтровать заголовки разделов
            filter_dates: Фильтровать даты и места с датами
            filter_phrases: Фильтровать служебные фразы
        """
        self.min_length = min_length
        self.filter_headers = filter_headers
        self.filter_dates = filter_dates
        self.filter_phrases = filter_phrases
    
    def is_non_informative(self, segment: str) -> bool:
        """
        Проверяет, является ли сегмент неинформативным.
        
        Args:
            segment: Текст сегмента
        
        Returns:
            True, если сегмент неинформативный и должен быть удален
        """
        segment = segment.strip()
        
        # Пустые сегменты
        if not segment:
            return True
        
        # Слишком короткие сегменты
        if len(segment) < self.min_length:
            return True
        
        # Проверка на служебные фразы
        if self.filter_phrases:
            segment_lower = segment.lower().strip()
            # Точное совпадение (без учета регистра)
            if segment_lower in self.NON_INFORMATIVE_PHRASES:
                return True
            
            # Проверка на фразы с двоеточием в конце (например, "Задачи:")
            if segment_lower.rstrip(":") in self.NON_INFORMATIVE_PHRASES:
                return True
        
        # Проверка на заголовки разделов
        if self.filter_headers:
            for pattern in self.SECTION_HEADER_PATTERNS:
                if re.match(pattern, segment, re.IGNORECASE):
                    return True
            
            # Проверка на заголовки вида "РАЗДЕЛ N. НАЗВАНИЕ" без дополнительного текста
            # Если после названия раздела нет содержательного текста
            section_match = re.match(r"^РАЗДЕЛ\s+\d+\.?\s+([А-ЯЁ\s]+)$", segment, re.IGNORECASE)
            if section_match:
                # Если после названия раздела только заголовок (все заглавные, короткий)
                title_part = section_match.group(1).strip()
                if len(title_part) < 50 and title_part.isupper():
                    # Проверяем, что это не полное предложение
                    if not any(char in segment for char in [".", ",", ":", ";"]):
                        return True
        
        # Проверка на даты
        if self.filter_dates:
            for pattern in self.DATE_PATTERNS:
                if re.match(pattern, segment):
                    return True
            
            for pattern in self.LOCATION_DATE_PATTERNS:
                if re.match(pattern, segment):
                    return True
        
        # Проверка на сегменты, состоящие только из цифр и служебных символов
        if re.match(r"^[\d\s\.\,\-\:]+$", segment):
            return True
        
        # Убрано: проверка на короткие заголовки заглавными буквами
        # Пользователь предпочитает оставлять такие сегменты
        
        return False
    
    def filter_segments(self, segments: List[str]) -> List[str]:
        """
        Фильтрует список сегментов, удаляя неинформативные.
        
        Args:
            segments: Список сегментов для фильтрации
        
        Returns:
            Отфильтрованный список сегментов
        """
        filtered = []
        for segment in segments:
            if not self.is_non_informative(segment):
                filtered.append(segment)
        return filtered

