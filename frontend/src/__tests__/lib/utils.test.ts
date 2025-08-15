import { cn, formatBytes, formatDate } from '@/lib/utils'

describe('utils', () => {
  describe('cn', () => {
    it('merges class names correctly', () => {
      expect(cn('class1', 'class2')).toBe('class1 class2')
    })

    it('handles conditional classes', () => {
      expect(cn('class1', false && 'class2', 'class3')).toBe('class1 class3')
    })

    it('handles Tailwind conflicts', () => {
      expect(cn('px-2', 'px-4')).toBe('px-4')
    })
  })

  describe('formatBytes', () => {
    it('formats bytes correctly', () => {
      expect(formatBytes(0)).toBe('0 Bytes')
      expect(formatBytes(1024)).toBe('1 KB')
      expect(formatBytes(1048576)).toBe('1 MB')
      expect(formatBytes(1073741824)).toBe('1 GB')
    })

    it('handles decimal places', () => {
      expect(formatBytes(1536, 1)).toBe('1.5 KB')
      expect(formatBytes(1536, 0)).toBe('2 KB')
    })
  })

  describe('formatDate', () => {
    it('formats date strings correctly', () => {
      const date = '2024-01-01T12:00:00Z'
      const formatted = formatDate(date)
      expect(formatted).toMatch(/Jan 1, 2024/)
    })

    it('formats Date objects correctly', () => {
      const date = new Date('2024-01-01T12:00:00Z')
      const formatted = formatDate(date)
      expect(formatted).toMatch(/Jan 1, 2024/)
    })

    it('includes time information', () => {
      const date = '2024-01-01T12:30:00Z'
      const formatted = formatDate(date)
      expect(formatted).toMatch(/30/)
    })
  })
})